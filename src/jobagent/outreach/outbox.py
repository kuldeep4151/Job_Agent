"""The outbox tick (`jobagent outbox`, every 15 minutes): read replies/bounces, queue follow-ups, send one.

Sending needs all of: outreach.enabled, live mode (not paused, not dry run), Gmail credentials, a weekday
inside send_hours (IST), and today's count under caps.emails_per_day. One email per tick keeps the pace human.
"""

import fcntl
import imaplib
import re
import smtplib
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlmodel import Session, select

from jobagent import health, settings
from jobagent.config import Config
from jobagent.discover.normalize import IST
from jobagent.models import Email, EmailStatus, Suppression, utcnow
from jobagent.outreach import compose, gmail
from jobagent.profile import Profile

DRAFT_MAX_AGE = timedelta(days=7)  # an unsent draft older than this is no longer a timely note
WATCH_REPLIES_FOR = timedelta(days=45)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def in_send_window(config: Config, now: datetime | None = None) -> bool:
    now = (now or datetime.now(IST)).astimezone(IST)
    start, end = config.outreach.send_hours
    return now.weekday() < 5 and start <= now.hour < end


def sent_today(session: Session) -> int:
    midnight = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    query = select(func.count()).select_from(Email).where(Email.status == EmailStatus.SENT, Email.sent_at >= midnight)
    return session.exec(query).one()


def is_suppressed(session: Session, address: str) -> str | None:
    for key in (address.lower(), "@" + address.lower().rpartition("@")[2]):
        if (row := session.get(Suppression, key)) is not None:
            return row.reason
    return None


def business_days_between(start: datetime, end: datetime) -> int:
    days, day = 0, start.astimezone(IST).date()
    while day < end.astimezone(IST).date():
        day += timedelta(days=1)
        days += day.weekday() < 5
    return days


def sending_blocked(session: Session, config: Config, now: datetime | None = None) -> str | None:
    """Why the outbox may not send right now, or None if it may."""
    if config.mode.paused:
        return "paused (mode.paused)"
    if config.mode.dry_run:
        return "dry run (mode.dry_run)"
    if not config.outreach.enabled:
        return "outreach is off (outreach.enabled: false); drafts wait in the dashboard"
    if gmail.credentials() is None:
        return "no Gmail App Password in .env"
    if not in_send_window(config, now):
        return f"outside sending hours (weekdays {config.outreach.send_hours[0]}:00-{config.outreach.send_hours[1]}:00 IST)"
    if sent_today(session) >= config.caps.emails_per_day:
        return f"daily email cap ({config.caps.emails_per_day}) reached"
    return None


# ---------- replies and bounces ----------

_BOUNCE_QUERY = "from:(mailer-daemon OR postmaster) newer_than:45d"


def check_inbox(session: Session, mailbox: gmail.Mailbox) -> list[str]:
    """Mark replies and bounces on sent emails; suppress bounced addresses and "no" replies."""
    events: list[str] = []
    since = utcnow() - WATCH_REPLIES_FOR
    watching = session.exec(select(Email).where(
        Email.status == EmailStatus.SENT, Email.sent_at >= since,
        Email.replied_at.is_(None), Email.bounced_at.is_(None),
    )).all()
    if not watching:
        return events

    bounce_texts = [mailbox.read(uid)[1].lower() for uid in mailbox.search(_BOUNCE_QUERY)]
    for sent in watching:
        address = sent.to_address.lower()
        if any(address in text for text in bounce_texts):
            sent.bounced_at = utcnow()
            session.merge(Suppression(address=address, reason="bounced"))
            events.append(f"bounced: {address}")
            continue
        domain = address.rpartition("@")[2]
        replies = mailbox.search(f"from:{domain} after:{_aware(sent.sent_at).astimezone(IST):%Y/%m/%d}")
        if replies:
            subject, text = mailbox.read(replies[-1])
            sent.replied_at = utcnow()
            sent.reply_snippet = (subject + " | " + " ".join(text.split())[:200]).strip(" |")
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if re.fullmatch(r"(no|no\.|no thanks|not interested)\W*", first_line, re.IGNORECASE):
                session.merge(Suppression(address=address, reason="asked not to be contacted"))
            events.append(f"reply from {domain}")
    session.commit()
    return events


# ---------- follow-ups and sending ----------


def queue_follow_ups(session: Session, config: Config, answers: dict, profile: Profile) -> int:
    queued = 0
    firsts = session.exec(select(Email).where(
        Email.kind == "first", Email.status == EmailStatus.SENT,
        Email.replied_at.is_(None), Email.bounced_at.is_(None),
    )).all()
    for first in firsts:
        if business_days_between(_aware(first.sent_at), utcnow()) < config.outreach.follow_up_after_days:
            continue
        already = session.exec(select(Email).where(Email.kind == "follow_up", Email.in_reply_to == first.message_id)).first()
        if already is None and not is_suppressed(session, first.to_address):
            session.add(compose.follow_up(first, answers, profile))
            queued += 1
    session.commit()
    return queued


def _replied_since(session: Session, email: Email) -> bool:
    """Any reply in this company's thread since the first email: don't send more."""
    return session.exec(select(Email).where(
        Email.company_key == email.company_key, Email.replied_at.is_not(None))).first() is not None


def send_next(session: Session, config: Config, profile: Profile, *, now: datetime | None = None,
              log: Callable[[str], None] = print) -> Email | None:
    if (reason := sending_blocked(session, config, now)) is not None:
        log(f"outbox: not sending ({reason})")
        return None
    drafts = session.exec(select(Email).where(Email.status == EmailStatus.DRAFT).order_by(Email.created_at)).all()
    for draft in drafts:
        if utcnow() - _aware(draft.created_at) > DRAFT_MAX_AGE and draft.kind == "first":
            draft.status, draft.note = EmailStatus.EXPIRED, "not sent within 7 days"
            continue
        if (why := is_suppressed(session, draft.to_address)) is not None:
            draft.status, draft.note = EmailStatus.CANCELLED, f"address suppressed ({why})"
            continue
        if _replied_since(session, draft):
            draft.status, draft.note = EmailStatus.CANCELLED, "the company already replied"
            continue
        sender, _password = gmail.credentials()
        attachment = settings.ROOT / draft.attachment if draft.attachment else None
        message = gmail.build_message(
            sender=sender, sender_name=profile.name, to=draft.to_address, subject=draft.subject,
            body=draft.body, attachment=attachment if attachment and attachment.exists() else None,
            in_reply_to=draft.in_reply_to,
        )
        try:
            gmail.send(message)
        except smtplib.SMTPAuthenticationError as e:
            log(f"outbox: Gmail rejected the App Password ({e.smtp_code}); nothing sent")
            session.commit()
            return None
        except (smtplib.SMTPException, OSError) as e:
            draft.status, draft.note = EmailStatus.FAILED, f"send failed: {e}"[:300]
            session.commit()
            log(f"outbox: sending to {draft.to_address} failed: {e}")
            return None
        draft.status, draft.sent_at, draft.message_id = EmailStatus.SENT, utcnow(), message["Message-ID"]
        session.commit()
        log(f"outbox: sent {draft.kind} email to {draft.to_address}: {draft.subject}")
        return draft
    session.commit()
    return None


LOCK_PATH = settings.LOG_DIR / "outbox.lock"


def tick(session: Session, config: Config, answers: dict, profile: Profile,
         log: Callable[[str], None] = print) -> None:
    """One outbox pass. Holds a file lock so a manual `jobagent outbox` and the scheduled one can never
    both pick the same draft and send it twice."""
    if config.mode.paused:
        log("outbox: paused")
        return
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("outbox: another outbox run is in progress; skipping")
            return
        _tick(session, config, answers, profile, log)


def _tick(session: Session, config: Config, answers: dict, profile: Profile, log: Callable[[str], None]) -> None:
    if not health.online():
        log("outbox: offline (Mac asleep?), nothing done")
        return
    if gmail.credentials() is not None and session.exec(select(Email).where(Email.status == EmailStatus.SENT)).first():
        try:
            with gmail.Mailbox() as mailbox:
                for event in check_inbox(session, mailbox):
                    log(f"outbox: {event}")
        except (imaplib.IMAP4.error, OSError) as e:
            log(f"outbox: couldn't read the inbox ({e}); will retry next tick")
    if queued := queue_follow_ups(session, config, answers, profile):
        log(f"outbox: queued {queued} follow-up(s)")
    if send_next(session, config, profile, log=log) is None and sending_blocked(session, config) is None:
        log("outbox: nothing to send")  # a line per tick, so outbox.log shows it's alive

