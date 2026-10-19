"""The main agent: every 15 minutes (`jobagent main`, launchd) it decides which sub-agents to run and lets
each work for a bounded time. Rule-based: it never calls Claude itself.

A tick: lock (never two at once) → paused? → online? → Finder, Applier, Mailer, Tracker, each only if
it's switched on, inside its hours, has work, and (for Claude work) today's budget isn't spent → one
AgentRun per turn → alerts → after 20:45, the daily report.
"""

import fcntl
import os
import signal
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlmodel import Session, select

from jobagent import alerts, db, digest, guards, health, llm, pipeline, settings
from jobagent.agents.applier import Applier
from jobagent.agents.base import SubAgent, Turn
from jobagent.agents.finder import Finder
from jobagent.agents.mailer import Mailer
from jobagent.agents.tracker import Tracker
from jobagent.discover.normalize import IST
from jobagent.match import MatchContext
from jobagent.models import AgentRun, utcnow
from jobagent.pipeline import Tally

AGENTS: list[SubAgent] = [Finder(), Applier(), Mailer(), Tracker()]  # in the order they take turns
LOCK_PATH = settings.LOG_DIR / "main.lock"
DIGEST_AT = (20, 45)
KEEP_AGENT_RUNS_DAYS = 30
TICK_LIMIT_MINUTES = 45  # a scheduled tick still running by then is stuck (seen live: a dead IMAP connection)


def by_name(name: str) -> SubAgent:
    return next(agent for agent in AGENTS if agent.name == name)


def summary(tally: Tally) -> str:
    return " · ".join(f"{name.replace('_', ' ')} {value}" for name, value in tally.counts().items()) or "nothing new"


def _claude_problem(match: MatchContext) -> str | None:
    if match.profile is None:
        return "config/profile.json is still the sample"
    if not llm.api_key_configured():
        return "no ANTHROPIC_API_KEY"
    if guards.budget_left_usd(match.config) <= 0:
        return "today's Claude budget is used up"
    return None


def waiting(agent: SubAgent, turn: Turn) -> str | None:
    """Why `agent` doesn't get a turn now, or None if it does."""
    if not agent.settings(turn.config).enabled:
        return f"off (agents.{agent.name}.enabled: false in config.yaml)"
    if turn.force:
        return None
    if why := agent.not_now(turn):
        return why
    queue = agent.queue(turn)
    if not queue:
        return agent.idle_note(turn)
    if set(queue) <= agent.claude_work and (problem := _claude_problem(turn.match)):
        return f"waiting for Claude: {problem}"
    return None


@contextmanager
def watchdog(minutes: float = TICK_LIMIT_MINUTES) -> Iterator[None]:
    """Stop a stuck tick, so it can't block every later one: first a TimeoutError in whatever hangs (recorded
    as that sub-agent's crash, and alerted), then, if that doesn't end it within 5 minutes, exit."""
    def stuck(_signum, _frame):
        signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
        signal.alarm(5 * 60)
        raise TimeoutError(f"still running after {minutes:g} minutes (stuck?); stopped so the next tick can start")

    previous = signal.signal(signal.SIGALRM, stuck)
    signal.alarm(max(1, int(minutes * 60)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def _lock() -> Iterator[bool]:
    with LOCK_PATH.open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def tick(*, force: bool = False, only: str | None = None, smoke: bool = False, now: datetime | None = None,
         log: Callable[[str], None] = print) -> Tally | None:
    """One tick. force: every (enabled) sub-agent gets a whole pass, ignoring hours and cadence
    (`jobagent run`); only: just that sub-agent. None when paused or another tick holds the lock."""
    match = pipeline.context()
    if match.config.mode.paused:
        log("Paused (mode.paused in config.yaml): nothing done.")
        return None
    with _lock() as mine:
        if not mine:
            log("Another tick (or `jobagent run`) is still working: skipped.")
            return None
        with db.session() as session:
            return _tick(session, match, force, only, smoke, (now or datetime.now(IST)).astimezone(IST), log)


def _tick(session: Session, match: MatchContext, force: bool, only: str | None, smoke: bool, now: datetime,
          log: Callable[[str], None]) -> Tally:
    record = AgentRun(agent="main")
    if not health.online():
        log(health.OFFLINE_NOTE)
        record.outcome, record.notes, record.finished_at = "offline", [health.OFFLINE_NOTE], utcnow()
        session.add(record)
        session.commit()
        return Tally(notes=[health.OFFLINE_NOTE])

    total, ran = Tally(), []
    for agent in AGENTS if only is None else [by_name(only)]:
        turn = Turn(session, match, now, until=float("inf"), force=force, smoke=smoke, log=log)
        if why := waiting(agent, turn):
            log(f"{agent.title}: {why}")
            continue
        if not force:
            turn.until = time.monotonic() + agent.minutes(match.config) * 60
        total.add(_take_turn(session, agent, turn))
        ran.append(agent.title)
    if not force and only is None:
        _daily_report(session, match, now, log)

    record.finished_at, record.counts = utcnow(), total.counts()
    record.summary = f"{', '.join(ran)}: {summary(total)}" if ran else "nothing to do"
    record.cost_usd = round(guards.spent_since(record.started_at), 4)
    session.add(record)
    session.commit()
    return total


def _take_turn(session: Session, agent: SubAgent, turn: Turn) -> Tally:
    tally, record = Tally(), AgentRun(agent=agent.name)
    turn.log(f"{agent.title}: starting")
    try:
        agent.run(turn, tally)
    except Exception as e:  # noqa: BLE001: one sub-agent's failure never stops the others
        session.rollback()
        tally.errors.append(f"crashed: {type(e).__name__}: {e}"[:400])
        turn.log(f"{agent.title} crashed:\n{traceback.format_exc()}")
    record.finished_at, record.counts, record.summary = utcnow(), tally.counts(), summary(tally)
    record.notes, record.errors = tally.notes, tally.errors
    record.outcome = "error" if any(e.startswith(health.FATAL) for e in tally.errors) else "ok"
    record.cost_usd = round(guards.spent_since(record.started_at), 4)
    session.add(record)
    session.commit()
    alerts.from_run(session, record)
    turn.log(f"{agent.title}: {record.summary}" + (f" (Claude ≈ ${record.cost_usd:.2f})" if record.cost_usd else ""))
    for note in tally.notes:
        turn.log(f"  note: {note}")
    for error in tally.errors:
        turn.log(f"  error: {error}")
    return tally


def _midnight_utc(now: datetime) -> datetime:
    return now.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def _daily_report(session: Session, match: MatchContext, now: datetime, log: Callable[[str], None]) -> None:
    """The 20:45 summary email, once a day (later if the Mac was asleep then), plus housekeeping."""
    if (now.hour, now.minute) < DIGEST_AT:
        return
    sent = select(AgentRun).where(AgentRun.agent == "digest", AgentRun.outcome == "ok",
                                  AgentRun.started_at >= _midnight_utc(now))
    if session.exec(sent).first() is not None:
        return
    record = AgentRun(agent="digest")
    try:
        record.summary = digest.send(session, match.config)
        if removed := digest.prune_screenshots(session):
            record.notes = [f"removed {removed} old screenshot(s)"]
        cutoff = utcnow() - timedelta(days=KEEP_AGENT_RUNS_DAYS)
        session.exec(delete(AgentRun).where(AgentRun.started_at < cutoff))
    except Exception as e:  # noqa: BLE001: retried next tick
        session.rollback()
        record.outcome, record.errors = "error", [f"crashed: {type(e).__name__}: {e}"[:400]]
    record.finished_at = utcnow()
    session.add(record)
    session.commit()
    log(f"Daily report: {record.summary or record.errors[0]}")


# ---------- status: `jobagent status` and the dashboard's Agents tab ----------


@dataclass
class AgentStatus:
    agent: SubAgent
    enabled: bool
    last: AgentRun | None
    queue: dict[str, int]
    waiting: str | None  # why it won't run on the next tick; None = it will


def status(session: Session, match: MatchContext, now: datetime | None = None) -> list[AgentStatus]:
    turn = Turn(session, match, (now or datetime.now(IST)).astimezone(IST), until=0)
    rows = []
    for agent in AGENTS:
        last = session.exec(select(AgentRun).where(AgentRun.agent == agent.name)
                            .order_by(AgentRun.started_at.desc())).first()
        why = "paused (mode.paused in config.yaml)" if match.config.mode.paused else waiting(agent, turn)
        rows.append(AgentStatus(agent, agent.settings(match.config).enabled, last, agent.queue(turn), why))
    return rows


def ticks_today(session: Session, now: datetime | None = None) -> list[AgentRun]:
    query = select(AgentRun).where(AgentRun.agent == "main",
                                   AgentRun.started_at >= _midnight_utc(now or datetime.now(IST)))
    return session.exec(query.order_by(AgentRun.started_at)).all()
