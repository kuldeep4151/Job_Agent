"""Is the agent working? Internet check before a tick, and the problems the daily report lists."""

import socket
from datetime import datetime, timezone

from sqlmodel import Session, select

from jobagent.config import Config
from jobagent.discover.normalize import IST
from jobagent.models import AgentRun, Run, aware

# Seen live: launchd fires overdue runs during macOS "dark wakes", when DNS and the screen are unavailable.
CHECK_HOST = ("api.anthropic.com", 443)
OFFLINE_NOTE = "Offline (no internet; the Mac was probably asleep): nothing done this run."
FATAL = ("crashed", "step ")  # errors that mean something broke (a search error on one site doesn't)
_WATCHED_NOTES = ("logged out", "couldn't start Chrome", "failures in a row")


def online(timeout: float = 6.0) -> bool:
    try:
        with socket.create_connection(CHECK_HOST, timeout=timeout):
            return True
    except OSError:
        return False


def problems(session: Session, config: Config, now: datetime | None = None) -> list[str]:
    """Plain-language problems for the daily report (empty = healthy). A problem that repeats every tick
    is listed once, with how often it happened."""
    now = (now or datetime.now(IST)).astimezone(IST)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    turns = session.exec(select(AgentRun).where(AgentRun.started_at >= midnight).order_by(AgentRun.started_at)).all()
    found: list[str] = []

    ticks = [t for t in turns if t.agent == "main"]
    if now.hour >= 12 and all(t.outcome == "offline" for t in ticks):
        found.append("No tick has run today. Is the Mac asleep, or the schedule off (`jobagent schedule status`)? "
                     "`sudo pmset repeat wakeorpoweron MTWRFSU 07:55:00` wakes it each morning.")
    seen: dict[str, list[datetime]] = {}
    for turn in turns:
        issues = [e[:160] for e in turn.errors if e.startswith(FATAL)]
        issues += [n for n in turn.notes if any(marker in n for marker in _WATCHED_NOTES)]
        for issue in issues:
            seen.setdefault(f"{turn.agent.title()}: {issue}", []).append(aware(turn.started_at))
    for text, times in seen.items():
        first = f"{times[0].astimezone(IST):%H:%M}"
        found.append(f"{text} (at {first})" if len(times) == 1 else f"{text} ({len(times)} times, first at {first})")

    sweeps = [r for r in session.exec(select(Run).order_by(Run.started_at.desc()).limit(10)).all()
              if r.finished_at and r.next_unit is None and OFFLINE_NOTE not in r.notes][:2]
    if len(sweeps) == 2:
        for site in config.search.sites:
            if all(not r.source_counts.get(site) for r in sweeps):
                found.append(f"No jobs from {site} in the last 2 searches (blocked, or the site changed).")
    return found
