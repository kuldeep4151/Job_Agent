"""Is the agent working? Internet check before a run, and the problems the digest and alerts report."""

import socket
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from jobagent.config import Config
from jobagent.discover.normalize import IST
from jobagent.models import Run

# Seen live: launchd fires overdue runs during macOS "dark wakes", when DNS and the screen are unavailable.
CHECK_HOST = ("api.anthropic.com", 443)
OFFLINE_NOTE = "Offline (no internet; the Mac was probably asleep): nothing done this run."


def online(timeout: float = 6.0) -> bool:
    try:
        with socket.create_connection(CHECK_HOST, timeout=timeout):
            return True
    except OSError:
        return False


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def runs_since(session: Session, since: datetime) -> list[Run]:
    return session.exec(select(Run).where(Run.started_at >= since).order_by(Run.started_at)).all()


def problems(session: Session, config: Config, now: datetime | None = None) -> list[str]:
    """Plain-language problems for the daily digest (empty = healthy)."""
    now = now or datetime.now(IST)
    midnight = now.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    today = runs_since(session, midnight.astimezone(timezone.utc))
    found: list[str] = []

    completed = [r for r in today if OFFLINE_NOTE not in r.notes]
    if now.astimezone(IST).hour >= 12 and not completed:
        found.append("No run has completed today. Is the Mac asleep at run times? "
                     "(`sudo pmset repeat wakeorpoweron MTWRFSU 07:55:00` wakes it for the first run.)")
    if offline := [r for r in today if OFFLINE_NOTE in r.notes]:
        found.append(f"{len(offline)} run(s) were skipped because the Mac had no internet (asleep).")
    for run in today:
        for error in run.errors:
            if error.startswith(("crashed", "step ")):
                found.append(f"Run at {_aware(run.started_at).astimezone(IST):%H:%M}: {error[:160]}")
        for note in run.notes:
            if "logged out" in note or "couldn't start Chrome" in note or "failures in a row" in note:
                found.append(f"Run at {_aware(run.started_at).astimezone(IST):%H:%M}: {note}")

    recent = [r for r in session.exec(select(Run).order_by(Run.started_at.desc()).limit(6)).all()
              if OFFLINE_NOTE not in r.notes][:2]
    if len(recent) == 2:
        for site in config.search.sites:
            if all(not r.source_counts.get(site) for r in recent):
                found.append(f"No jobs from {site} in the last 2 runs (blocked, or the site changed).")
    return list(dict.fromkeys(found))
