"""The daily summary email to yourself (sent by the main agent after 20:45, or `jobagent digest`): replies
first, then what each sub-agent did today, what needs you, spend, and health."""

import time
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import func
from sqlmodel import Session, select

from jobagent import guards, health, settings
from jobagent.config import Config
from jobagent.discover.normalize import IST
from jobagent.models import AgentRun, Application, Email, EmailStatus, Job, State, aware
from jobagent.outreach import gmail

DASHBOARD = "http://localhost:8501"
NEEDS_YOU_SHOWN = 5
SCREENSHOT_DAYS = 60


_NAMES = {"naukri": "Naukri", "linkedin": "LinkedIn", "indeed": "Indeed", "glassdoor": "Glassdoor",
          "greenhouse": "Greenhouse", "lever": "Lever", "smartrecruiters": "SmartRecruiters", "icims": "iCIMS",
          "successfactors": "SuccessFactors", "company site": "company site"}


def _where(job: Job) -> str:
    where = job.ats or job.source
    return _NAMES.get(where, where.title())


def _n(count: int, word: str, plural: str | None = None) -> str:
    return f"{count} {word if count == 1 else plural or word + 's'}"


def build(session: Session, config: Config, now: datetime | None = None) -> tuple[str, str]:
    now = (now or datetime.now(IST)).astimezone(IST)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)

    turns = session.exec(select(AgentRun).where(AgentRun.started_at >= midnight).order_by(AgentRun.started_at)).all()
    ticks = [t for t in turns if t.agent == "main"]
    offline = sum(t.outcome == "offline" for t in ticks)
    did: dict[str, Counter] = {}
    for turn in turns:
        did.setdefault(turn.agent, Counter()).update(turn.counts)
    finder, tracker = did.get("finder", Counter()), did.get("tracker", Counter())
    statuses = Counter(session.exec(select(Application.status)).all())
    applied = session.exec(select(Job).where(Job.applied_at >= midnight).order_by(Job.applied_at)).all()
    needs_you = session.exec(select(Job).where(Job.state == State.MANUAL).order_by(Job.score.desc())).all()
    sent = session.exec(select(Email).where(Email.status == EmailStatus.SENT, Email.sent_at >= midnight)).all()
    replies = session.exec(select(Email).where(Email.replied_at >= midnight)).all()
    bounces = session.exec(select(Email).where(Email.bounced_at >= midnight)).all()
    drafts = session.exec(select(func.count()).select_from(Email).where(Email.status == EmailStatus.DRAFT)).one()
    problems = health.problems(session, config, now)
    companies = {j.id: j.company for j in session.exec(select(Job).where(Job.id.in_([e.job_id for e in replies]))).all()}

    lines = [f"Job agent · {now:%A %d %B}", ""]
    if replies:
        lines.append(f"REPLIES ({len(replies)})")
        for e in replies:
            lines.append(f"  • {companies.get(e.job_id, e.to_address)} ({e.to_address}): {e.reply_snippet or ''}"[:220])
        lines.append("")

    lines.append("TODAY")
    last = f", last at {aware(ticks[-1].started_at).astimezone(IST):%H:%M}" if ticks else ""
    lines.append(f"  Main agent: {_n(len(ticks) - offline, 'tick')}{last}"
                 + (f" ({offline} more found no internet: Mac asleep)" if offline else ""))
    lines.append(f"  Finder: {_n(finder['searched'], 'search', 'searches')}, {finder['found']} jobs found "
                 f"({finder['new']} new), {finder['scored']} scored, {_n(finder['matches'], 'match', 'matches')}, "
                 f"{_n(finder['tailored'], 'resume')} tailored")
    lines.append(f"  Applier: applied to {len(applied)}" + (":" if applied else " today"))
    lines += [f"    • {j.company} · {j.title} ({_where(j)})" for j in applied]
    lines.append(f"  Mailer: {len(sent)} sent, {drafts} draft(s) waiting"
                 + (f", {len(bounces)} bounced" if bounces else "")
                 + ("" if config.outreach.enabled else " (sending is off)"))
    if statuses:
        changes = tracker["tracked"] + tracker["updated"]
        lines.append(f"  Tracker: {_n(sum(statuses.values()), 'application')} · "
                     + ", ".join(f"{n} {status}" for status, n in statuses.most_common())
                     + (f" ({_n(changes, 'change')} today)" if changes else ""))
    lines.append(f"  Claude: ${guards.spent_today_usd():.2f} today (cap ${config.caps.usd_per_day:.2f}) · "
                 f"${guards.spent_this_month_usd():.2f} this month (cap ${config.caps.usd_per_month:.2f})")
    lines.append("")

    if needs_you:
        lines.append(f"NEEDS YOU ({len(needs_you)}, best first)")
        for j in needs_you[:NEEDS_YOU_SHOWN]:
            lines.append(f"  • {j.score or '–'} {j.company} · {j.title} · {_where(j)}")
            lines.append(f"    {DASHBOARD}/?job={j.id}")
        if len(needs_you) > NEEDS_YOU_SHOWN:
            lines.append(f"  … and {len(needs_you) - NEEDS_YOU_SHOWN} more in the dashboard's \"Needs you\" tab.")
        lines.append("")

    lines.append("HEALTH")
    lines += [f"  ⚠ {p}" for p in problems] or ["  ✓ All good."]
    lines += ["", f"Dashboard: `uv run jobagent dashboard`, then {DASHBOARD}"]

    subject = f"Job agent · {now:%a %d %b}: {len(applied)} applied, {len(needs_you)} need you"
    if replies:
        subject = f"Job agent · {len(replies)} repl{'y' if len(replies) == 1 else 'ies'}! " + subject.split(": ", 1)[1]
    if problems:
        subject += f" · ⚠ {len(problems)} issue{'s' if len(problems) > 1 else ''}"
    return subject, "\n".join(lines)


def prune_screenshots(session: Session, days: int = SCREENSHOT_DAYS) -> int:
    """Delete evidence older than `days`, except screenshots still referenced by a job."""
    keep = set(session.exec(select(Job.apply_screenshot).where(Job.apply_screenshot.is_not(None))).all())
    cutoff, removed = time.time() - days * 86_400, 0
    for folder in (settings.LOG_DIR / "applications", settings.LOG_DIR / "naukri"):
        for path in folder.glob("*.*") if folder.exists() else []:
            stored = settings.stored_path(path)
            if path.stat().st_mtime < cutoff and stored not in keep and stored.removesuffix(".html") + ".png" not in keep:
                path.unlink()
                removed += 1
    return removed


def send(session: Session, config: Config) -> str:
    creds = gmail.credentials()
    subject, body = build(session, config)
    if creds is None:
        return "no Gmail App Password in .env: digest not sent"
    gmail.send(gmail.build_message(sender=creds[0], sender_name="Job Agent", to=creds[0], subject=subject, body=body))
    return f"sent: {subject}"
