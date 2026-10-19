"""Finder: searches the job boards, screens what it finds (hard filters, skill prefilter, Claude score),
tailors a resume for each match that needs one, and routes it to where it's applied.

A sweep (every role × city × site) is searched in slices: each turn searches for at most 60% of its
time, then screens. The cursor lives on the sweep's Run row, so a Mac that sleeps mid-sweep loses one
slice, not hours, and the Applier and Mailer get their turns in between.
"""

import time
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import or_
from sqlmodel import Session, select

from jobagent import health, pipeline
from jobagent.agents.base import SubAgent, Turn, count, non_zero
from jobagent.config import Config, load_targets
from jobagent.discover import ats_boards, jobspy_src
from jobagent.discover.normalize import IST
from jobagent.models import Job, Run, State, aware, utcnow

SEARCH_SHARE = 0.6  # of a turn, at most, for searching: the rest screens what's been found


def open_sweep(session: Session) -> Run | None:
    return session.exec(select(Run).where(Run.next_unit.is_not(None)).order_by(Run.id.desc())).first()


def last_sweep(session: Session) -> Run | None:
    """The most recent sweep (runs from before the sub-agents count too; offline/crashed ones don't)."""
    for run in session.exec(select(Run).order_by(Run.started_at.desc()).limit(20)).all():
        if health.OFFLINE_NOTE not in run.notes and not any(e.startswith("crashed") for e in run.errors):
            return run
    return None


def next_sweep_at(session: Session, config: Config) -> datetime | None:
    """When the next sweep may start (UTC); None = now."""
    last = last_sweep(session)
    if last is None:
        return None
    due = aware(last.started_at) + timedelta(hours=config.agents.finder.sweep_every_hours)
    return due if due > utcnow() else None


class Finder(SubAgent):
    name, title = "finder", "Finder"
    claude_work = frozenset({"to score", "to tailor"})

    def queue(self, turn: Turn) -> dict[str, int]:
        session, config = turn.session, turn.config
        total = len(jobspy_src.units(config.search))
        sweep = open_sweep(session)
        searches = max(total - sweep.next_unit, 1) if sweep else (total if next_sweep_at(session, config) is None else 0)
        return non_zero({
            "searches left": searches,
            "to filter": count(session, select(Job).where(Job.state == State.DISCOVERED)),
            "to score": count(session, select(Job).where(Job.state == State.FILTERED)),
            "to tailor": count(session, select(Job).where(  # Naukri's own apply sends the profile, not a PDF
                Job.state == State.SCORED, or_(Job.source != "naukri", Job.apply_url.is_not(None)))),
            "to route": count(session, select(Job).where(Job.state == State.TAILORED, Job.ats.is_(None))),
        })

    def idle_note(self, turn: Turn) -> str:
        due = next_sweep_at(turn.session, turn.config)
        return f"nothing to do; next search at {due.astimezone(IST):%H:%M}" if due else "nothing to do"

    def run(self, turn: Turn, tally: pipeline.Tally) -> None:
        session, ctx, log = turn.session, turn.match, turn.log
        started = time.monotonic()
        sweep = None if turn.smoke else open_sweep(session)
        if sweep is None and (turn.force or next_sweep_at(session, turn.config) is None):
            sweep = Run(next_unit=0)
            session.add(sweep)
            session.commit()
        if sweep is not None:
            search_until = started + SEARCH_SHARE * (turn.until - started)
            pipeline.step("search", session, tally, lambda: self._search(turn, sweep, tally, search_until), log)
        steps = [
            ("filter", lambda: pipeline.filter_discovered(session, ctx, tally)),
            ("score", lambda: pipeline.score_filtered(session, ctx, tally, until=turn.until)),
            ("tailor", lambda: pipeline.tailor_scored(session, ctx, tally, until=turn.until)),
            ("route", lambda: pipeline.route_tailored(session, tally)),
        ]
        for name, action in steps:
            pipeline.step(name, session, tally, action, log)

    def _search(self, turn: Turn, sweep: Run, tally: pipeline.Tally, until: float) -> None:
        session, search = turn.session, turn.config.search
        total, start = len(jobspy_src.units(search, turn.smoke)), sweep.next_unit
        turn.log(f"Searching {', '.join(search.sites)}: from search {start + 1} of {total}…")
        batch = jobspy_src.Batch()
        sweep.next_unit = jobspy_src.discover(search, batch, turn.smoke, start=start, until=until)
        if sweep.next_unit is None and not turn.smoke:
            ats_boards.discover(load_targets(), search, batch)
        new, duplicates = pipeline.store(session, batch.jobs)

        sweep.found, sweep.new, sweep.duplicates = sweep.found + len(batch.jobs), sweep.new + new, sweep.duplicates + duplicates
        sweep.source_counts = dict(Counter(sweep.source_counts) + batch.counts)
        sweep.errors = [*sweep.errors, *batch.errors]
        tally.searched += (total if sweep.next_unit is None else sweep.next_unit) - start
        tally.found, tally.new, tally.duplicates = tally.found + len(batch.jobs), tally.new + new, tally.duplicates + duplicates
        tally.source_counts.update(batch.counts)
        tally.errors += batch.errors
        if sweep.next_unit is None:
            sweep.finished_at = utcnow()
            sweep.notes = [f"No jobs from {site} this sweep." for site in search.sites if not sweep.source_counts.get(site)]
            tally.notes += sweep.notes
        session.commit()
        progress = "sweep done" if sweep.next_unit is None else f"{sweep.next_unit} of {total} searches done"
        turn.log(f"Found {len(batch.jobs)} ({new} new, {duplicates} duplicates); {progress}.")
