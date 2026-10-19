"""Immediate alert emails to yourself, for problems that need you: at most one per kind per day."""

from datetime import datetime, timezone

from sqlmodel import Session, select

from jobagent.discover.normalize import IST
from jobagent.models import AgentRun, Alert
from jobagent.outreach import gmail

# note fragment → (alert kind, what to do)
_FROM_NOTES = [
    ("Naukri is logged out", "naukri_logged_out", "Run `uv run jobagent naukri login` to sign in again."),
    ("Naukri logged out mid-run", "naukri_logged_out", "Run `uv run jobagent naukri login` to sign in again."),
    ("failures in a row", "naukri_failing", "Naukri's pages may have changed: check the screenshots in the dashboard."),
]


def notify(session: Session, kind: str, message: str) -> bool:
    """Email `message` to yourself unless this kind was already alerted today. Returns True if sent."""
    midnight = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    if session.exec(select(Alert).where(Alert.kind == kind, Alert.created_at >= midnight)).first():
        return False
    alert = Alert(kind=kind, message=message)
    creds = gmail.credentials()
    if creds is not None:
        try:
            gmail.send(gmail.build_message(sender=creds[0], sender_name="Job Agent", to=creds[0],
                                           subject=f"Job agent alert: {message.splitlines()[0][:80]}", body=message))
            alert.emailed = True
        except Exception:  # an alert must never break the run that raised it
            pass
    session.add(alert)
    session.commit()
    return alert.emailed


def from_run(session: Session, run: AgentRun) -> list[str]:
    """Alerts a sub-agent's finished turn calls for; returns the kinds raised."""
    raised = []
    for fragment, kind, advice in _FROM_NOTES:
        if any(fragment in note for note in run.notes):
            notify(session, kind, f"{fragment}. {advice}")
            raised.append(kind)
    if crashes := [e for e in run.errors if e.startswith(("crashed", "step "))]:
        who = "The main agent" if run.agent == "main" else f"The {run.agent.title()}"
        notify(session, "run_errors", f"{who} hit an error: {crashes[0][:300]}\n\nSee data/logs/main.log.")
        raised.append("run_errors")
    return raised
