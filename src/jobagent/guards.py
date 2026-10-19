"""Limits that keep a fully automatic agent safe: daily Claude spend, live/dry mode, daily apply caps."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from sqlalchemy import func
from sqlmodel import Session, select

from jobagent import llm, setup_status
from jobagent.config import Config
from jobagent.discover.normalize import IST
from jobagent.models import Job

ApplyMode = Literal["off", "dry", "live"]


def apply_mode(config: Config) -> tuple[ApplyMode, str]:
    """Live only when the user asked for it AND every setup item is real (no placeholders)."""
    if config.mode.paused:
        return "off", "paused (mode.paused in config.yaml)"
    if config.mode.dry_run:
        return "dry", "dry run (mode.dry_run in config.yaml)"
    if not setup_status.ready_for_live():
        return "dry", "setup incomplete, so dry run only (see `jobagent check`)"
    return "live", ""


def _start_of_today_utc() -> datetime:
    """Midnight IST, in UTC (SQLModel requires timezone-aware datetimes in queries)."""
    midnight = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.astimezone(timezone.utc)


def naukri_attempts_today(session: Session, live: bool) -> int:
    """Live: applications submitted today. Dry: dry-run attempts today (so a dry run mirrors live volume)."""
    column = Job.applied_at if live else Job.dry_run_at
    query = select(func.count()).select_from(Job).where(Job.source == "naukri", column >= _start_of_today_utc())
    return session.exec(query).one()


def _spent_since(start, log_path: Path | None = None) -> float:
    path = log_path or llm.USAGE_LOG
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if datetime.fromisoformat(record["ts"]).astimezone(IST).date() >= start:
            total += record["cost_usd"]
    return total


def spent_today_usd(log_path: Path | None = None) -> float:
    """Estimated Claude spend since midnight IST, from the usage log."""
    return _spent_since(datetime.now(IST).date(), log_path)


def spent_this_month_usd(log_path: Path | None = None) -> float:
    return _spent_since(datetime.now(IST).date().replace(day=1), log_path)


def budget_left_usd(config: Config) -> float:
    """What may still be spent now: the tighter of today's and this month's caps."""
    return min(config.caps.usd_per_day - spent_today_usd(), config.caps.usd_per_month - spent_this_month_usd())


def claude_available(config: Config) -> bool:
    return llm.api_key_configured() and budget_left_usd(config) > 0


def spent_since(moment: datetime, log_path: Path | None = None) -> float:
    """Claude spend since an exact moment (one sub-agent's turn), from the usage log."""
    path = log_path or llm.USAGE_LOG
    if not path.exists():
        return 0.0
    records = (json.loads(line) for line in path.read_text().splitlines())
    return sum(r["cost_usd"] for r in records if datetime.fromisoformat(r["ts"]) >= moment)


def ats_attempts_today(session: Session, live: bool) -> int:
    """Greenhouse/Lever applications (or dry runs) today: their own cap, separate from Naukri's."""
    column = Job.applied_at if live else Job.dry_run_at
    query = select(func.count()).select_from(Job).where(Job.ats.in_(["greenhouse", "lever"]), column >= _start_of_today_utc())
    return session.exec(query).one()
