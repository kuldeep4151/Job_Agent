"""Applier: fills and submits application forms (Naukri's own apply, Greenhouse, Lever) for the matches
the Finder routed to it, within the daily caps, live only when config and setup allow it.

Jobs it can't do by itself are already in "Needs you" (the Finder's routing); a form with a question
answers.yaml can't answer goes there too, with the question noted.
"""

from datetime import timedelta

from sqlmodel import Session, select

from jobagent import guards, pipeline
from jobagent.agents.base import SubAgent, Turn, count, non_zero
from jobagent.models import AgentRun, utcnow

LOGGED_OUT_RETRY = timedelta(hours=2)  # don't pop Chrome open every tick while Naukri waits for your login


def naukri_logged_out(session: Session) -> bool:
    """Did a turn in the last 2 hours (the Applier's or the Tracker's) find Naukri logged out?"""
    recent = session.exec(select(AgentRun).where(AgentRun.agent.in_(["applier", "tracker"]),
                                                 AgentRun.started_at >= utcnow() - LOGGED_OUT_RETRY)).all()
    return any("logged out" in note for run in recent for note in run.notes)


class Applier(SubAgent):
    name, title = "applier", "Applier"
    claude_work = frozenset({"naukri", "greenhouse/lever"})  # form answers come from Claude

    def not_now(self, turn: Turn) -> str | None:
        start, end = self.settings(turn.config).hours
        if not start <= turn.now.hour < end:
            return f"outside its hours ({start:02d}:00-{end:02d}:00 IST)"
        mode, why = guards.apply_mode(turn.config)
        if mode == "off":
            return why
        if turn.match.profile is None:
            return "config/profile.json is still the sample"
        return None

    def queue(self, turn: Turn) -> dict[str, int]:
        """Matches it may still apply to today: ready jobs, up to what's left of each daily cap."""
        session, caps = turn.session, turn.config.caps
        live = guards.apply_mode(turn.config)[0] == "live"
        naukri_room = caps.naukri_applies_per_day - guards.naukri_attempts_today(session, live)
        ats_room = caps.ats_applies_per_day - guards.ats_attempts_today(session, live)
        naukri = 0 if naukri_logged_out(session) else min(count(session, pipeline.naukri_ready(live)), naukri_room)
        return non_zero({
            "naukri": naukri,
            "greenhouse/lever": min(count(session, pipeline.ats_ready(live)), ats_room),
        })

    def idle_note(self, turn: Turn) -> str:
        if naukri_logged_out(turn.session):
            return "Naukri is logged out: run `jobagent naukri login` (it retries every 2 hours)"
        return "nothing to do (or today's caps are reached)"

    def run(self, turn: Turn, tally: pipeline.Tally) -> None:
        session, ctx, log = turn.session, turn.match, turn.log
        if turn.force or not naukri_logged_out(session):
            pipeline.step("naukri", session, tally,
                          lambda: pipeline.apply_naukri(session, ctx, tally, until=turn.until, log=log), log)
        pipeline.step("greenhouse/lever", session, tally,
                      lambda: pipeline.apply_ats(session, ctx, tally, until=turn.until, log=log), log)
