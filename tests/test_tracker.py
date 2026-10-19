"""The Tracker: the list of applications, their statuses, replies read by Claude (mocked), the CSV."""

import csv
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlmodel import select

from conftest import SAMPLE_PROFILE
from jobagent import db, guards, pipeline
from jobagent.agents import tracker
from jobagent.agents.base import Turn
from jobagent.apply import naukri
from jobagent.config import Config, Search
from jobagent.discover.normalize import IST
from jobagent.models import AgentRun, Application, AppStatus, Email, EmailStatus, Job, State, utcnow
from jobagent.outreach import gmail
from jobagent.profile import load_profile

NOW = datetime.now(IST)


@pytest.fixture
def world(monkeypatch, tmp_path, fake_api):
    db.configure(tmp_path / "jobs.db")
    config = Config(reviewed=True, search=Search(roles=["x"], cities=["Pune"]))
    monkeypatch.setattr(pipeline, "load_config", lambda: config)
    monkeypatch.setattr(pipeline, "load_answers", lambda: {})
    monkeypatch.setattr(pipeline, "_real_profile", lambda: load_profile(SAMPLE_PROFILE))
    monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcdefghijklmnop")
    sent: list = []
    monkeypatch.setattr(gmail, "send", sent.append)
    return config, fake_api, sent


def job(i: int, company: str, state: str = State.APPLIED, **fields) -> Job:
    fields.setdefault("source", "naukri")
    fields.setdefault("external_id", str(i))
    return Job(id=i, dedupe_key=f"k{i}", url=f"https://jobs.example/{i}", title="ML Engineer", company=company,
               state=state, **fields)


def email(job_id: int, to: str, status: str = EmailStatus.SENT, **fields) -> Email:
    return Email(job_id=job_id, company_key=to.split("@")[1], to_address=to, contact_source="job post",
                 subject="ML Engineer - Sample Candidate", body="b", status=status, sent_at=utcnow(), **fields)


def add(*rows) -> None:
    with db.session() as session:
        session.add_all(rows)
        session.commit()


def applications() -> dict[str, Application]:
    with db.session() as session:
        return {app.company: app for app in session.exec(select(Application)).all()}


def take_turn(config) -> pipeline.Tally:
    tally = pipeline.Tally()
    with db.session() as session:
        tracker.Tracker().run(Turn(session, pipeline.context(), now=NOW, until=float("inf")), tally)
    return tally


def test_every_application_is_listed_once(world):
    yesterday = utcnow() - timedelta(days=1)
    add(job(1, "Acme", applied_at=yesterday),
        job(2, "Orbit", source="linkedin", ats="greenhouse", applied_at=yesterday),
        job(3, "Nimbus", applied_at=yesterday, apply_note="applied by you (marked in the dashboard)"),
        job(4, "Pixel", State.MANUAL), email(4, "hr@pixel.in"),
        job(5, "Delta"),  # Naukri said "Applied" before the agent got there
        job(6, "Echo", State.SCORED), email(6, "hr@echo.in", EmailStatus.DRAFT))  # not sent: not on the list
    with db.session() as session:
        assert tracker.sync_applications(session) == 5
        assert tracker.sync_applications(session) == 0
    apps = applications()
    assert {c: a.via for c, a in apps.items()} == {"Acme": "Naukri", "Orbit": "Greenhouse", "Nimbus": "you",
                                                    "Pixel": "email", "Delta": "Naukri"}
    assert apps["Pixel"].emailed_to == "hr@pixel.in" and apps["Delta"].applied_at is None
    assert apps["Acme"].history[0]["status"] == "applied" and apps["Acme"].history[0]["by"] == "Naukri"

    add(email(1, "careers@acme.in"))  # emailed after applying
    with db.session() as session:
        session.get(Job, 4).state = State.APPLIED  # applied by hand after the email
        session.get(Job, 4).apply_note = "applied by you (marked in the dashboard)"
        session.commit()
        assert tracker.sync_applications(session) == 0
    apps = applications()
    assert apps["Acme"].emailed_to == "careers@acme.in" and apps["Pixel"].via == "you"


def test_automatic_updates_only_move_forward():
    app = Application(job_id=1, company="Acme", title="t", via="Naukri")
    assert tracker.set_status(app, AppStatus.VIEWED, "Naukri")
    assert not tracker.set_status(app, AppStatus.APPLIED, "Naukri")
    assert tracker.set_status(app, AppStatus.INTERVIEW, "email reply", "Invites to a call")
    assert not tracker.set_status(app, AppStatus.VIEWED, "Naukri")  # a late "viewed" never undoes it
    assert not tracker.set_status(app, AppStatus.NO_RESPONSE, "Tracker")
    assert tracker.set_status(app, AppStatus.REJECTED, "email reply", "Position filled")
    assert not tracker.set_status(app, AppStatus.INTERVIEW, "email reply")  # settled: only you reopen it
    assert tracker.set_status(app, AppStatus.INTERVIEW, tracker.YOU, "they called back")
    assert [h["status"] for h in app.history] == ["viewed", "interview", "rejected", "interview"]
    assert (app.status, app.status_note) == ("interview", "they called back")


@pytest.mark.parametrize(("kind", "status"), [("interview", "interview"), ("rejection", "rejected"),
                                              ("needs_info", "replied"), ("auto_reply", "applied")])
def test_replies_are_read_and_move_the_application(world, kind, status):
    config, api, sent = world
    add(job(1, "Acme", applied_at=utcnow() - timedelta(days=2)),
        email(1, "hr@acme.in", replied_at=utcnow(), reply_snippet="Re: ML Engineer | Are you free on Tuesday?",
              reply_text="Subject: Re: ML Engineer\n\nAre you free on Tuesday at 11 for an interview?"))
    api.reply(json.dumps({"kind": kind, "summary": "Asks for an interview on Tuesday"}))

    tally = take_turn(config)

    app = applications()["Acme"]
    assert app.status == status
    with db.session() as session:
        assert session.exec(select(Email)).one().reply_kind == kind
    assert "Are you free on Tuesday at 11" in api.sent[0]["body"]["messages"][0]["content"]
    alerts = [m for m in sent if m["Subject"].startswith("Job agent alert: Interview invite from Acme")]
    assert len(alerts) == (kind == "interview")
    assert tally.tracked == 1 and tally.updated == (kind != "auto_reply")

    take_turn(config)
    assert len(api.sent) == 1  # each reply is read once


def test_replies_wait_for_the_claude_budget(world, monkeypatch):
    config, api, _ = world
    monkeypatch.setattr(guards, "budget_left_usd", lambda config: 0)
    add(job(1, "Acme"), email(1, "hr@acme.in", replied_at=utcnow(), reply_snippet="Re: hi | Let's talk"))
    tally = take_turn(config)
    assert not api.sent and applications()["Acme"].status == "applied"
    assert any("waits for Claude" in note for note in tally.notes)


def test_a_bounce_is_noted_once(world):
    config, _, _ = world
    add(job(1, "Acme", State.MANUAL), email(1, "hr@acme.in", bounced_at=utcnow()))
    assert take_turn(config).counts() == {"tracked": 1, "updated": 1}
    acme = applications()["Acme"]
    assert (acme.status, acme.status_note) == ("applied", "email to hr@acme.in bounced")
    with db.session() as session:
        assert tracker.Tracker().queue(Turn(session, pipeline.context(), now=NOW, until=0)) == {}


def test_no_news_for_weeks_becomes_no_response(world):
    config, _, _ = world
    weeks_ago, days_ago = utcnow() - timedelta(days=22), utcnow() - timedelta(days=10)
    add(Application(job_id=1, company="Quiet", title="t", via="Naukri", status_at=weeks_ago),
        Application(job_id=2, company="Recent", title="t", via="Naukri", status_at=days_ago),
        Application(job_id=3, company="Talking", title="t", via="Naukri", status="interview", status_at=weeks_ago))
    with db.session() as session:
        assert tracker.mark_no_response(session, config) == 1
    assert {c: a.status for c, a in applications().items()} == {
        "Quiet": "no response", "Recent": "applied", "Talking": "interview"}


def test_csv_lists_applications_newest_first(world):
    config, _, _ = world
    add(job(1, "Old", applied_at=utcnow() - timedelta(days=3)), job(2, "New", applied_at=utcnow()), job(3, "Earlier"))
    take_turn(config)
    with tracker.CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == tracker.CSV_COLUMNS
    assert [r["Company"] for r in rows] == ["New", "Old", "Earlier"]
    assert rows[0]["Applied on"] == f"{NOW:%Y-%m-%d}" and rows[2]["Applied on"] == "earlier"
    assert rows[0]["Status"] == "applied" and rows[0]["Link"] == "https://jobs.example/2"


def test_tracker_queue(world):
    config, _, _ = world
    add(job(1, "Acme"), email(1, "hr@acme.in", replied_at=utcnow()))
    with db.session() as session:
        turn = Turn(session, pipeline.context(), now=NOW, until=0)
        assert tracker.Tracker().queue(turn) == {"to add": 1, "replies to read": 1}
        tracker.sync_applications(session)
        session.get(Email, 1).reply_kind = "other"
        session.commit()
        assert tracker.Tracker().queue(turn) == {"csv": 1}
        tracker.export_csv(session)
        assert tracker.Tracker().queue(turn) == {}


def test_dashboard_lists_applications_and_saves_your_status(world):
    from test_pipeline import open_dashboard

    add(job(1, "Acme", applied_at=utcnow()))
    with db.session() as session:
        tracker.sync_applications(session)
    app = open_dashboard(1)
    assert not app.exception, app.exception
    assert any("Tracker" in m.value for m in app.markdown)
    app.selectbox(key="status-1").set_value("interview")
    app.text_input(key="status-note-1").input("HR round on Friday")
    app.button(key="status-save-1").click().run()
    assert not app.exception, app.exception
    acme = applications()["Acme"]
    assert (acme.status, acme.status_note, acme.history[-1]["by"]) == ("interview", "HR round on Friday", "you")
    assert "interview" in tracker.CSV_PATH.read_text()


# ---------- Naukri's application-status page ----------

APPLIES = Path(__file__).parent / "fixtures" / "naukri" / "applies.html"


def test_read_applies_from_the_status_page(monkeypatch):
    from playwright.sync_api import sync_playwright

    monkeypatch.setattr(naukri, "APPLIES_URL", APPLIES.as_uri())
    with sync_playwright() as pw:
        browser_ = pw.chromium.launch()
        page = browser_.new_page()
        page.wait_for_timeout = lambda ms: None  # no settling pauses for a local file
        applies = naukri.read_applies(page)
        browser_.close()
    assert [(a.job_id, a.company, a.status) for a in applies] == [
        ("111111111111", "Acme", "Application sent today"),
        ("222222222222", "Orbit", "Application viewed 3 days ago"),
        ("333333333333", "Nimbus", "Not shortlisted"),
    ]


@pytest.fixture
def naukri_page(world, monkeypatch):
    """The Tracker opening Chrome on the status page, served from the fixture."""
    config, _, _ = world
    config.agents.tracker.naukri_sync_hours = [10, 18]
    state = {"logged_in": True, "opened": 0}

    @contextmanager
    def fake_chrome():
        state["opened"] += 1
        yield SimpleNamespace(pages=["page"])

    monkeypatch.setattr(tracker.browser, "naukri_context", fake_chrome)
    monkeypatch.setattr(naukri, "is_logged_in", lambda page: state["logged_in"])
    monkeypatch.setattr(naukri, "read_applies", lambda page: naukri.parse_applies(APPLIES.read_text()))
    for i, (company, naukri_id) in enumerate([("Acme", "111111111111"), ("Orbit", "222222222222"),
                                               ("Nimbus", "333333333333"), ("Delta", "444444444444")], 1):
        add(job(i, company, external_id=f"nk-{naukri_id}", applied_at=utcnow()))
    return config, state


def test_naukri_statuses_move_applications_on(naukri_page):
    config, state = naukri_page
    tally = pipeline.Tally()
    with db.session() as session:
        tracker.sync_applications(session)
        tracker.check_naukri(session, tally)
    apps = applications()
    assert {c: (a.status, a.status_note) for c, a in apps.items()} == {
        "Acme": ("applied", None), "Orbit": ("viewed", "Application viewed"),
        "Nimbus": ("rejected", "Not shortlisted"), "Delta": ("applied", None)}  # Delta isn't on the page
    assert (tally.checked, tally.updated, state["opened"]) == (1, 2, 1)
    assert apps["Orbit"].history[-1]["by"] == "Naukri"


def test_unknown_naukri_chips_are_noted_not_guessed():
    assert tracker.naukri_chip("Application sent 2 days ago") == (True, None)
    assert tracker.naukri_chip("Application viewed by recruiter") == (True, AppStatus.VIEWED)
    assert tracker.naukri_chip("Recruiter sent you a message") == (False, None)


def test_naukri_check_runs_at_its_hours_and_backs_off_when_logged_out(naukri_page):
    config, state = naukri_page

    def turn_at(hour: int, minute: int = 0) -> dict:
        with db.session() as session:
            turn = Turn(session, pipeline.context(), now=NOW.replace(hour=hour, minute=minute), until=float("inf"))
            agent = tracker.Tracker()
            queue = agent.queue(turn)
            if queue:
                record = AgentRun(agent="tracker")
                tally = pipeline.Tally()
                agent.run(turn, tally)
                record.counts, record.notes = tally.counts(), tally.notes
                session.add(record)
                session.commit()
            return queue

    assert "naukri check" not in turn_at(9, 30)
    assert turn_at(10, 5)["naukri check"] == 1 and state["opened"] == 1
    assert "naukri check" not in turn_at(10, 20)  # done for the 10:00 slot

    state["logged_in"] = False
    with db.session() as session:  # pretend the 10:00 check never happened, so the 18:00 one is the next
        for run in session.exec(select(AgentRun)).all():
            session.delete(run)
        session.commit()
    turn_at(18, 5)
    assert state["opened"] == 2
    assert "naukri check" not in turn_at(18, 20)  # logged out: no Chrome every 15 minutes
    assert state["opened"] == 2
