"""Mailer: all the email work. Finds a hiring contact for each recent match (the job post, the company's
website, a verified web lookup), drafts one email, reads replies and bounces, queues one follow-up, and
sends at most one email per turn inside the sending hours (see outreach/outbox.py for every rule).
"""

from sqlmodel import select

from jobagent import pipeline
from jobagent.agents.base import SubAgent, Turn, count, non_zero
from jobagent.models import Email, EmailStatus, utcnow
from jobagent.outreach import gmail, outbox


class Mailer(SubAgent):
    name, title = "mailer", "Mailer"
    claude_work = frozenset({"to contact"})

    def not_now(self, turn: Turn) -> str | None:
        return "config/profile.json is still the sample" if turn.match.profile is None else None

    def queue(self, turn: Turn) -> dict[str, int]:
        session, config = turn.session, turn.config
        drafts = count(session, select(Email).where(Email.status == EmailStatus.DRAFT))
        room = config.caps.emails_per_day - drafts  # never more drafts than a day's sending
        watching = select(Email).where(
            Email.status == EmailStatus.SENT, Email.sent_at >= utcnow() - outbox.WATCH_REPLIES_FOR,
            Email.replied_at.is_(None), Email.bounced_at.is_(None))
        return non_zero({
            "to contact": min(count(session, pipeline.outreach_candidates(config)), room),
            "to send": drafts if outbox.sending_blocked(session, config) is None else 0,
            "watching": count(session, watching) if gmail.credentials() else 0,  # replies, bounces, follow-ups
        })

    def run(self, turn: Turn, tally: pipeline.Tally) -> None:
        session, ctx, log = turn.session, turn.match, turn.log
        pipeline.step("contacts", session, tally,
                      lambda: pipeline.draft_outreach(session, ctx, tally, until=turn.until, log=log), log)
        if ctx.profile is not None:
            pipeline.step("outbox", session, tally, lambda: self._outbox(turn, tally), log)

    def _outbox(self, turn: Turn, tally: pipeline.Tally) -> None:
        result = outbox.tick(turn.session, turn.config, turn.match.answers, turn.match.profile, log=turn.log)
        tally.replies += sum(event.startswith("reply") for event in result.events)
        tally.bounced += sum(event.startswith("bounced") for event in result.events)
        tally.follow_ups += result.follow_ups
        tally.sent += result.sent is not None
