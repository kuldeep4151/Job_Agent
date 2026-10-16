"""Phase 7: health checks, alerts, the digest, budget caps, housekeeping."""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from jobagent import alerts, db, digest, guards, health, settings
from jobagent.config import Config, Search
from jobagent.discover.normalize import IST
from jobagent.models import Alert, Email, EmailStatus, Job, Run, State, utcnow
from jobagent.outreach import gmail

NOON = datetime(2026, 10, 16, 13, 0, tzinfo=IST)


@pytest.fixture
def world(monkeypatch, tmp_path):
    db.configure(tmp_path / "jobs.db")
    monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcdefghijklmnop")
    sent: list = []
    monkeypatch.setattr(gmail, "send", sent.append)
    config = Config(reviewed=True, search=Search(roles=["x"], cities=["Pune"], sites=["naukri", "linkedin"]))
    return config, sent


def add(*rows):
    with db.session() as session:
        for row in rows:
            session.add(row)
        session.commit()


def at(hour: int, minute: int = 0) -> datetime:
    return NOON.replace(hour=hour, minute=minute).astimezone(timezone.utc)


def test_healthy_day_has_no_problems(world):
    config, _ = world
    add(Run(started_at=at(8), source_counts={"naukri": 10, "linkedin": 4}),
        Run(started_at=at(11), source_counts={"naukri": 8, "linkedin": 2}))
    with db.session() as session:
        assert health.problems(session, config, NOON) == []


def test_problems_are_explained(world):
    config, _ = world
    add(Run(started_at=at(8), notes=[health.OFFLINE_NOTE]),
        Run(started_at=at(9), source_counts={"naukri": 5}, notes=["Naukri is logged out: run `jobagent naukri login`."]),
        Run(started_at=at(11), source_counts={"naukri": 3}, errors=["step score failed: RuntimeError: boom"]))
    with db.session() as session:
        found = health.problems(session, config, NOON)
    assert "1 run(s) were skipped because the Mac had no internet (asleep)." in found
    assert any("09:00" in p and "logged out" in p for p in found)
    assert any("11:00" in p and "step score failed" in p for p in found)
    assert "No jobs from linkedin in the last 2 runs (blocked, or the site changed)." in found


def test_no_completed_run_by_noon(world):
    config, _ = world
    with db.session() as session:
        assert any(p.startswith("No run has completed today") for p in health.problems(session, config, NOON))


def test_alerts_are_emailed_once_per_kind_per_day(world):
    _, sent = world
    run = Run(notes=["Naukri is logged out: run `jobagent naukri login`. Nothing was applied."])
    with db.session() as session:
        assert alerts.from_run(session, run) == ["naukri_logged_out"]
        alerts.from_run(session, run)
        assert len(session.exec(select(Alert)).all()) == 1
    assert len(sent) == 1 and sent[0]["Subject"].startswith("Job agent alert: Naukri is logged out")
    assert sent[0]["To"] == "me@gmail.com"


def test_alert_failure_never_breaks_the_run(world, monkeypatch):
    def broken(_message):
        raise OSError("network down")
    monkeypatch.setattr(gmail, "send", broken)
    with db.session() as session:
        alerts.notify(session, "x", "something")  # no exception
        assert session.exec(select(Alert)).one().emailed is False


def test_digest_puts_replies_first_and_lists_what_needs_you(world):
    config, _ = world
    add(Job(id=1, dedupe_key="a", source="naukri", external_id="1", url="u", title="AI Trainee", company="Acme",
            state=State.APPLIED, applied_at=at(9)),
        Job(id=2, dedupe_key="b", source="linkedin", external_id="2", url="u", title="ML Engineer", company="Orbit",
            score=78, state=State.MANUAL, ats="linkedin"),
        Email(job_id=1, company_key="acme", to_address="hr@acme.in", contact_source="job post", subject="s",
              body="b", status=EmailStatus.SENT, sent_at=at(10), replied_at=at(12), reply_snippet="Re: s | Let's talk"),
        Run(started_at=at(8), found=40, new=12, scored=5, matches=1, source_counts={"naukri": 30, "linkedin": 10}))
    with db.session() as session:
        subject, body = digest.build(session, config, NOON)
    assert subject == "Job agent · 1 reply! 1 applied, 1 need you"
    assert body.index("REPLIES (1)") < body.index("TODAY") < body.index("NEEDS YOU")
    assert "Acme (hr@acme.in): Re: s | Let's talk" in body
    assert "• Acme · AI Trainee (Naukri)" in body and "Jobs: 40 found, 12 new, 5 scored, 1 match" in body
    assert "• 78 Orbit · ML Engineer · LinkedIn" in body and f"{digest.DASHBOARD}/?job=2" in body
    assert "✓ All good." in body


def test_monthly_cap_also_limits_spend(world, monkeypatch, tmp_path):
    config, _ = world
    log = tmp_path / "usage.jsonl"
    early_this_month = datetime.now(IST).replace(day=1, hour=1).astimezone(timezone.utc)
    log.write_text(json.dumps({"ts": early_this_month.isoformat(), "task": "score", "cost_usd": 59.5}) + "\n")
    monkeypatch.setattr(guards.llm, "USAGE_LOG", log)
    assert guards.budget_left_usd(config) == pytest.approx(0.5)  # month cap 60 is tighter than the day's 4.00


def test_log_rotation(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LOG_DIR", tmp_path)
    (tmp_path / "outbox.log").write_text("x" * 100)
    settings.rotate_logs(max_bytes=50)
    assert (tmp_path / "outbox.log.1").exists() and not (tmp_path / "outbox.log").exists()


def test_old_unreferenced_screenshots_are_pruned(world, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LOG_DIR", tmp_path)
    folder = tmp_path / "applications"
    folder.mkdir()
    old, kept, recent = folder / "job-1-old.png", folder / "job-2-applied.png", folder / "job-3-new.png"
    for path in (old, kept, recent):
        path.write_bytes(b"png")
    sixty_one_days_ago = time.time() - 61 * 86_400
    for path in (old, kept):
        os.utime(path, (sixty_one_days_ago, sixty_one_days_ago))
    add(Job(dedupe_key="k", source="naukri", external_id="2", url="u", title="t", company="c",
            apply_screenshot=str(kept)))
    with db.session() as session:
        assert digest.prune_screenshots(session) == 1
    assert not old.exists() and kept.exists() and recent.exists()
