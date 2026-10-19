"""The main agent and its sub-agents: who gets a turn, deadlines, the lock, the daily report, and the
Finder's sweep in slices. Fake sub-agents stand in where the real ones' work isn't the point."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlmodel import select

from conftest import SAMPLE_PROFILE
from jobagent import db, guards, pipeline
from jobagent.agents import applier, finder, main, mailer
from jobagent.agents.base import SubAgent, Turn
from jobagent.config import Config, Search
from jobagent.discover import jobspy_src
from jobagent.discover.normalize import IST
from jobagent.models import AgentRun, Email, EmailStatus, Job, Run, State, utcnow
from jobagent.outreach import gmail, outbox
from jobagent.profile import load_profile

TODAY = datetime.now(IST).replace(hour=10, minute=0, second=0, microsecond=0)


class Fake(SubAgent):
    def __init__(self, name: str, queue: dict | None = None, *, not_now: str | None = None,
                 claude_work: set[str] = frozenset(), crash: bool = False):
        self.name, self.title = name, name.title()
        self._queue, self._not_now, self.claude_work, self.crash = queue or {}, not_now, frozenset(claude_work), crash
        self.turns: list[Turn] = []

    def not_now(self, turn):
        return self._not_now

    def queue(self, turn):
        return self._queue

    def run(self, turn, tally):
        self.turns.append(turn)
        if self.crash:
            raise RuntimeError("boom")
        tally.applied += 1


@pytest.fixture
def world(monkeypatch, tmp_path, fake_api):
    db.configure(tmp_path / "jobs.db")
    config = Config(reviewed=True, search=Search(roles=["Backend Developer"], cities=["Pune"], include_remote=False,
                                                 sites=["naukri"]))
    monkeypatch.setattr(pipeline, "load_config", lambda: config)
    monkeypatch.setattr(pipeline, "load_answers", lambda: {})
    monkeypatch.setattr(pipeline, "_real_profile", lambda: load_profile(SAMPLE_PROFILE))
    return config


def use(monkeypatch, *agents: SubAgent) -> None:
    monkeypatch.setattr(main, "AGENTS", list(agents))


def turns() -> list[AgentRun]:
    with db.session() as session:
        return session.exec(select(AgentRun).order_by(AgentRun.id)).all()


def test_only_agents_with_work_get_a_turn(world, monkeypatch):
    world.agents.tracker.enabled = False
    finder_, applier_ = Fake("finder", {"to score": 2}), Fake("applier", {"naukri": 1}, not_now="outside its hours")
    mailer_, tracker_ = Fake("mailer"), Fake("tracker", {"to add": 1})
    use(monkeypatch, finder_, applier_, mailer_, tracker_)
    logs: list[str] = []

    total = main.tick(now=TODAY, log=logs.append)

    assert len(finder_.turns) == 1 and not (applier_.turns or mailer_.turns or tracker_.turns)
    assert total.applied == 1
    assert {"Applier: outside its hours", "Mailer: nothing to do",
            "Tracker: off (agents.tracker.enabled: false in config.yaml)"} <= set(logs)
    assert [(t.agent, t.outcome, t.summary) for t in turns()] == [
        ("finder", "ok", "applied 1"), ("main", "ok", "Finder: applied 1")]


def test_run_gives_every_enabled_agent_a_whole_pass(world, monkeypatch):
    world.agents.mailer.enabled = False
    agents = [Fake("finder"), Fake("applier", {"naukri": 1}, not_now="outside its hours"), Fake("mailer")]
    use(monkeypatch, *agents)
    main.tick(force=True, now=TODAY, log=lambda _: None)
    assert [len(a.turns) for a in agents] == [1, 1, 0]
    assert agents[0].turns[0].until == float("inf")


def test_claude_work_waits_for_the_budget(world, monkeypatch):
    monkeypatch.setattr(guards, "budget_left_usd", lambda config: 0)
    finder_ = Fake("finder", {"to score": 3, "to filter": 1}, claude_work={"to score"})  # filtering is free
    applier_ = Fake("applier", {"naukri": 1}, claude_work={"naukri"})
    use(monkeypatch, finder_, applier_)
    logs: list[str] = []
    main.tick(now=TODAY, log=logs.append)
    assert len(finder_.turns) == 1 and not applier_.turns
    assert "Applier: waiting for Claude: today's Claude budget is used up" in logs


def test_a_crash_is_recorded_alerted_and_the_rest_still_run(world, monkeypatch):
    monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcdefghijklmnop")
    sent: list = []
    monkeypatch.setattr(gmail, "send", sent.append)
    broken, fine = Fake("finder", {"x": 1}, crash=True), Fake("applier", {"naukri": 1})
    use(monkeypatch, broken, fine)

    total = main.tick(now=TODAY, log=lambda _: None)

    assert len(fine.turns) == 1 and total.errors == ["crashed: RuntimeError: boom"]
    finder_turn = turns()[0]
    assert (finder_turn.agent, finder_turn.outcome) == ("finder", "error")
    assert len(sent) == 1 and "The Finder hit an error: crashed: RuntimeError: boom" in sent[0].get_content()


def test_each_turn_gets_its_own_time_budget(world, monkeypatch):
    world.agents.applier.minutes_per_tick = 2
    agent = Fake("applier", {"naukri": 1})
    use(monkeypatch, agent)
    clock = SimpleNamespace(monotonic=lambda: 1000.0)
    monkeypatch.setattr(main, "time", clock)
    main.tick(now=TODAY, log=lambda _: None)
    assert agent.turns[0].until == 1120.0


def test_one_tick_at_a_time(world, monkeypatch):
    agent = Fake("finder", {"x": 1})
    use(monkeypatch, agent)
    logs: list[str] = []
    with main._lock() as mine:
        assert mine
        assert main.tick(now=TODAY, log=logs.append) is None
    assert not agent.turns and not turns() and "still working" in logs[0]
    assert main.tick(now=TODAY, log=logs.append) is not None  # lock released


def test_a_stuck_turn_is_stopped_and_the_tick_finishes(world, monkeypatch):
    import time as real_time

    class Stuck(Fake):
        def run(self, turn, tally):
            real_time.sleep(30)  # a dead connection

    stuck, after = Stuck("finder", {"x": 1}), Fake("applier", {"naukri": 1})
    use(monkeypatch, stuck, after)
    started = real_time.monotonic()
    with main.watchdog(minutes=1 / 60):
        total = main.tick(now=TODAY, log=lambda _: None)
    assert real_time.monotonic() - started < 5
    assert total.errors == ["crashed: TimeoutError: still running after 0.0166667 minutes (stuck?); "
                            "stopped so the next tick can start"]
    assert len(after.turns) == 1  # the rest of the tick still ran


def test_paused_does_nothing_and_records_nothing(world, monkeypatch):
    world.mode.paused = True
    agent = Fake("finder", {"x": 1})
    use(monkeypatch, agent)
    assert main.tick(now=TODAY, log=lambda _: None) is None
    assert not agent.turns and not turns()


def test_daily_report_once_after_2045(world, monkeypatch):
    monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcdefghijklmnop")
    sent: list = []
    monkeypatch.setattr(gmail, "send", sent.append)
    use(monkeypatch)
    for hour, minute in [(20, 30), (20, 50), (21, 5)]:
        main.tick(now=TODAY.replace(hour=hour, minute=minute), log=lambda _: None)
    assert len(sent) == 1 and sent[0]["Subject"].startswith("Job agent ·")
    assert [t.agent for t in turns()] == ["main", "digest", "main", "main"]
    main.tick(force=True, now=TODAY.replace(hour=22), log=lambda _: None)  # `jobagent run` never sends it
    assert len(sent) == 1


def test_status_explains_each_agent(world, monkeypatch):
    use(monkeypatch, Fake("finder", {"to score": 2}), Fake("applier", {"naukri": 1}, not_now="outside its hours"))
    with db.session() as session:
        rows = main.status(session, pipeline.context(), now=TODAY)
    assert [(r.agent.name, r.queue, r.waiting) for r in rows] == [
        ("finder", {"to score": 2}, None), ("applier", {"naukri": 1}, "outside its hours")]


# ---------- the Finder: a sweep in slices ----------


def posting(i: int) -> dict:
    return {"id": str(i), "site": "naukri", "job_url": f"https://naukri.example/{i}", "title": "Backend Developer",
            "company": f"Co {i}", "location": "Chennai, India", "description": "Python.", "date_posted": None}


@pytest.fixture
def slow_boards(world, monkeypatch):
    """4 searches (2 roles × 2 sites) that each take 100 seconds on a fake clock."""
    world.search = Search(roles=["A", "B"], cities=["Pune"], include_remote=False, sites=["naukri", "linkedin"])
    world.agents.finder.minutes_per_tick = 5  # 300 s a turn, so 180 s for searching
    clock = {"now": 0.0}
    fake_time = SimpleNamespace(monotonic=lambda: clock["now"], sleep=lambda seconds: None)
    for module in (main, finder, jobspy_src, pipeline):
        monkeypatch.setattr(module, "time", fake_time)
    searches: list[tuple[str, str]] = []

    def scrape(site_name, search_term, **_kwargs):
        searches.append((site_name[0], search_term))
        clock["now"] += 100
        return pd.DataFrame([posting(len(searches))])

    monkeypatch.setattr(jobspy_src, "scrape_jobs", scrape)
    monkeypatch.setattr(finder.ats_boards, "discover", lambda *a: None)
    use(monkeypatch, finder.Finder())
    return searches


def sweeps() -> list[Run]:
    with db.session() as session:
        return session.exec(select(Run).order_by(Run.id)).all()


def test_finder_searches_in_slices_and_resumes(slow_boards):
    main.tick(now=TODAY, log=lambda _: None)
    assert slow_boards == [("naukri", "A"), ("linkedin", "A")]  # the third would start past 180 s
    (sweep,) = sweeps()
    assert (sweep.next_unit, sweep.found, sweep.finished_at) == (2, 2, None)

    main.tick(now=TODAY, log=lambda _: None)
    assert slow_boards[2:] == [("naukri", "B"), ("linkedin", "B")]
    (sweep,) = sweeps()
    assert sweep.next_unit is None and sweep.finished_at is not None
    assert (sweep.found, sweep.new, sweep.source_counts) == (4, 4, {"naukri": 2, "linkedin": 2})
    finder_turns = [t for t in turns() if t.agent == "finder"]
    assert [t.counts["searched"] for t in finder_turns] == [2, 2]
    assert finder_turns[1].counts["skipped"] == 2  # screened in the same turn: Chennai isn't on the list
    with db.session() as session:
        assert {j.state for j in session.exec(select(Job)).all()} == {State.SKIPPED}


def test_next_sweep_waits_for_its_cadence(slow_boards):
    main.tick(force=True, now=TODAY, log=lambda _: None)  # a whole sweep at once
    assert len(slow_boards) == 4 and sweeps()[0].next_unit is None
    logs: list[str] = []
    main.tick(now=TODAY, log=logs.append)
    assert len(slow_boards) == 4 and any(line.startswith("Finder: nothing to do; next search at") for line in logs)

    with db.session() as session:  # 3 hours later
        sweep = session.exec(select(Run)).one()
        sweep.started_at = utcnow() - timedelta(hours=3, minutes=1)
        session.commit()
    main.tick(now=TODAY, log=lambda _: None)
    assert len(slow_boards) == 6 and sweeps()[-1].next_unit == 2


# ---------- the Applier's and Mailer's queues ----------


def naukri_match(i: int, **fields) -> Job:
    return Job(dedupe_key=f"n{i}", source="naukri", external_id=str(i), url=f"https://naukri.example/{i}",
               title="Backend Developer", company=f"Co {i}", state=State.SCORED, score=80, **fields)


def test_applier_queue_follows_caps_hours_and_login(world):
    world.caps.naukri_applies_per_day = 2
    with db.session() as session:
        session.add_all([naukri_match(1), naukri_match(2), naukri_match(3), naukri_match(4, dry_run_at=utcnow())])
        session.commit()
        turn = Turn(session, pipeline.context(), now=TODAY, until=0)
        agent = applier.Applier()
        assert agent.queue(turn) == {"naukri": 1}  # 3 ready, but a dry run already used 1 of today's 2
        assert agent.not_now(turn) is None
        turn.now = TODAY.replace(hour=22)
        assert agent.not_now(turn) == "outside its hours (09:00-21:00 IST)"

        session.add(AgentRun(agent="applier", notes=["Naukri is logged out: run `jobagent naukri login`."]))
        session.commit()
        assert agent.queue(turn) == {}  # no Chrome popping up every 15 minutes until you log in
        assert "jobagent naukri login" in agent.idle_note(turn)


def test_mailer_counts_what_the_outbox_did(world, monkeypatch):
    sent = Email(job_id=1, company_key="acme", to_address="hr@acme.in", contact_source="job post", subject="s",
                 body="b", status=EmailStatus.SENT)
    result = outbox.TickResult(events=["reply from acme.in", "bounced: x@y.in"], follow_ups=1, sent=sent)
    monkeypatch.setattr(outbox, "tick", lambda *a, **k: result)
    monkeypatch.setattr(pipeline, "draft_outreach", lambda *a, **k: [])
    tally = pipeline.Tally()
    with db.session() as session:
        mailer.Mailer().run(Turn(session, pipeline.context(), now=TODAY, until=float("inf")), tally)
    assert tally.counts() == {"sent": 1, "replies": 1, "bounced": 1, "follow_ups": 1}
