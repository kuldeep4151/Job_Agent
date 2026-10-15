"""One run: discover → store (dedupe) → hard filters → skill prefilter → Claude score.

Each stage only picks up jobs in its input state, so a crashed or interrupted run is
finished by the next one, and nothing is scored twice.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import anthropic
from playwright.sync_api import Error as PlaywrightError
from sqlalchemy import func
from sqlmodel import Session, select

from jobagent import db, guards, llm, settings
from jobagent.apply import browser, naukri
from jobagent.apply.answers import AnswerMapper
from jobagent.config import load_answers, load_config, load_targets
from jobagent.discover import ats_boards, jobspy_src
from jobagent.discover.jobspy_src import Batch
from jobagent.discover.normalize import norm_company
from jobagent.match import (
    MatchContext,
    blocklisted,
    apply_score,
    hard_filter,
    is_stale,
    matching_skills,
    score_job,
    scoring_system_prompt,
)
from jobagent.models import Email, EmailStatus, Job, Run, State, utcnow
from jobagent.outreach import compose, contacts, outbox
from jobagent.profile import Profile, load_profile
from jobagent.tailor import tailor


NO_DESCRIPTION = "returned no job description"


@dataclass
class Tally:
    skipped: int = 0
    scored: int = 0
    matches: int = 0
    tailored: int = 0
    applied: int = 0
    drafted: int = 0
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


def store(session: Session, jobs: list[Job]) -> tuple[int, int]:
    """Insert new jobs; for duplicates, keep the first row and merge in the extra URL/emails."""
    new = duplicates = 0
    for job in jobs:
        existing = session.exec(select(Job).where(Job.dedupe_key == job.dedupe_key)).first()
        if existing is None:
            session.add(job)
            session.flush()
            new += 1
            continue
        duplicates += 1
        if job.url != existing.url and job.url not in existing.also_seen_at:
            existing.also_seen_at = [*existing.also_seen_at, job.url]
        extra_emails = [e for e in job.emails if e not in existing.emails]
        if extra_emails:
            existing.emails = [*existing.emails, *extra_emails]
        if not existing.apply_url and job.apply_url:
            existing.apply_url = job.apply_url
        if not existing.description.strip() and job.description.strip():
            existing.description = job.description  # another board had the text the first one withheld
            if existing.state == State.SKIPPED and (existing.skip_reason or "").endswith(NO_DESCRIPTION):
                existing.state, existing.skip_reason = State.DISCOVERED, None
    session.commit()
    return new, duplicates


def filter_discovered(session: Session, ctx: MatchContext, tally: Tally) -> None:
    for job in session.exec(select(Job).where(Job.state == State.DISCOVERED)).all():
        reason = hard_filter(job, ctx)
        if reason:
            job.state, job.skip_reason = State.SKIPPED, reason
            tally.skipped += 1
        else:
            job.state = State.FILTERED
    session.commit()


def score_filtered(session: Session, ctx: MatchContext, tally: Tally) -> None:
    if ctx.profile is None:
        tally.notes.append("Scoring skipped: config/profile.json is still the sample placeholder.")
        return
    has_key = llm.api_key_configured()
    if not has_key:
        tally.notes.append("Claude scoring skipped: no ANTHROPIC_API_KEY (skill prefilter still ran).")
    system = scoring_system_prompt(ctx.profile, ctx.answers)
    skills = ctx.profile.all_skills()
    minimum = ctx.config.matching.min_skill_overlap

    pending = session.exec(
        select(Job).where(Job.state == State.FILTERED).order_by(Job.first_seen_at.desc())
    ).all()
    for job in pending:  # freshest first, so a budget stop leaves only older jobs waiting
        if is_stale(job):
            job.state, job.skip_reason = State.SKIPPED, "stale: first seen over 72h ago"
            tally.skipped += 1
        elif not job.description.strip():
            job.state, job.skip_reason = State.SKIPPED, f"{job.source} {NO_DESCRIPTION}"
            tally.skipped += 1
        elif len(overlap := matching_skills(job, skills)) < minimum:
            job.skill_overlap = overlap
            job.state, job.skip_reason = State.SKIPPED, f"only {len(overlap)} of your skills mentioned"
            tally.skipped += 1
        elif not has_key:
            job.skill_overlap = overlap
        elif guards.budget_left_usd(ctx.config) <= 0:
            tally.note(f"Daily Claude budget (${ctx.config.caps.usd_per_day}) reached; rest waits.")
            break
        else:
            job.skill_overlap = overlap
            try:
                apply_score(job, score_job(job, system), ctx.config.matching.min_score)
            except llm.LLMError as e:
                job.state, job.skip_reason = State.SKIPPED, f"scoring failed: {e}"
                tally.errors.append(f"job {job.id}: {e}")
                tally.skipped += 1
            except anthropic.APIError as e:  # SDK already retried; stop and leave the rest for next run
                tally.errors.append(f"Claude API error, scoring stopped: {e}")
                break
            else:
                tally.scored += 1
                tally.matches += job.state == State.SCORED
        session.commit()


def tailor_scored(session: Session, ctx: MatchContext, tally: Tally) -> None:
    """Per-job resume for every match that won't go through Naukri's own (profile-based) apply."""
    if ctx.profile is None or not llm.api_key_configured():
        return  # score_filtered already noted why
    system = tailor.tailoring_system_prompt(ctx.profile)
    pending = session.exec(select(Job).where(Job.state == State.SCORED).order_by(Job.score.desc())).all()
    for job in pending:  # best matches first, so a budget stop hits the weakest
        if tailor.uses_naukri_profile(job):
            continue
        if guards.budget_left_usd(ctx.config) <= 0:
            tally.note(f"Daily Claude budget (${ctx.config.caps.usd_per_day}) reached; rest waits.")
            break
        failure = None
        try:
            result = tailor.build_tailored(ctx.profile, tailor.plan_for(job, system))
        except llm.LLMError as e:
            result, failure = None, f"tailoring failed ({e}); using your original resume"
            tally.errors.append(f"job {job.id}: {e}")
        except anthropic.APIError as e:
            tally.errors.append(f"Claude API error, tailoring stopped: {e}")
            break
        tailor.save(job, ctx.profile, result, failure)
        tally.tailored += 1
        session.commit()


PAUSE_SECONDS = (20.0, 60.0)  # between Naukri applications: human pace, fewer bot flags


def _record_outcome(job: Job, outcome: naukri.Outcome) -> None:
    job.apply_note = outcome.note
    if outcome.screenshot:
        job.apply_screenshot = settings.stored_path(outcome.screenshot)
    job.apply_qa = naukri.qa_json(outcome) or job.apply_qa
    if outcome.status == "applied":
        job.state, job.applied_at = State.APPLIED, utcnow()
    elif outcome.status == "dry_run":
        job.dry_run_at = utcnow()
    elif outcome.status == "already_applied":
        job.state = State.APPLIED  # applied before, not by this run: applied_at stays empty
    elif outcome.status == "company_site":
        job.apply_url = job.url  # the "Apply on company site" button lives on this page → tailored next run
    elif outcome.status == "closed":
        job.state, job.skip_reason = State.SKIPPED, "closed on Naukri"
    elif outcome.status == "manual":
        job.state = State.MANUAL
    elif outcome.status == "failed":
        job.state = State.FAILED


def apply_naukri(
    session: Session, ctx: MatchContext, tally: Tally, *, job_id: int | None = None, log: Callable[[str], None] = print
) -> None:
    """Apply (or dry-run) to the best Naukri matches, within today's cap, in the logged-in Chrome profile."""
    mode, why = guards.apply_mode(ctx.config)
    if mode == "off" or ctx.profile is None:
        return
    live = mode == "live"
    if job_id is not None:
        query = select(Job).where(Job.id == job_id, Job.source == "naukri")
    else:
        query = select(Job).where(Job.state == State.SCORED, Job.source == "naukri", Job.apply_url.is_(None))
        query = query.where(Job.applied_at.is_(None) if live else Job.dry_run_at.is_(None))
    remaining = ctx.config.caps.naukri_applies_per_day - guards.naukri_attempts_today(session, live)
    if remaining <= 0:
        tally.note(f"Naukri daily cap ({ctx.config.caps.naukri_applies_per_day}) reached.")
        return
    jobs = session.exec(query.order_by(Job.score.desc()).limit(remaining)).all()
    if not jobs:
        return
    if not live:
        tally.note(f"Naukri: {why}; nothing was submitted.")

    mapper = AnswerMapper(ctx.answers, ctx.profile)
    with browser.naukri_context() as context:
        page = browser.first_page(context)
        if not naukri.is_logged_in(page):
            tally.note("Naukri is logged out: run `jobagent naukri login`. Nothing was applied.")
            return
        failures_in_a_row = 0
        for i, job in enumerate(jobs):
            if i:
                time.sleep(random.uniform(*PAUSE_SECONDS))
            log(f"Naukri {'apply' if live else 'dry run'}: {job.title} at {job.company}")
            try:
                outcome = naukri.apply_to_job(page, job, mapper, live=live)
            except PlaywrightError as e:
                outcome = naukri.Outcome("failed", f"browser error: {e}"[:300])
                try:
                    outcome.screenshot = naukri.save_evidence(page, job.id, "error", html=True)
                except PlaywrightError:
                    pass  # the page itself is gone; the note still says what happened
            _record_outcome(job, outcome)
            session.commit()
            if outcome.status == "logged_out":
                tally.note("Naukri logged out mid-run: run `jobagent naukri login`.")
                break
            tally.applied += outcome.status == "applied"
            broken = outcome.status == "failed" or outcome.note.startswith("couldn't recognise")
            failures_in_a_row = failures_in_a_row + 1 if broken else 0
            if failures_in_a_row >= 2:  # Naukri probably changed its pages: stop before doing damage
                tally.note("Stopped Naukri after 2 failures in a row; check the screenshots in the dashboard.")
                break


OUTREACH_STATES = (State.SCORED, State.TAILORED, State.APPLIED, State.MANUAL)
OUTREACH_MAX_AGE = timedelta(days=7)


def draft_outreach(session: Session, ctx: MatchContext, tally: Tally, *, job_id: int | None = None,
                   log: Callable[[str], None] = print) -> list[Email]:
    """Find a hiring contact for recent matches and draft one email each (the outbox sends them later)."""
    config = ctx.config
    if config.mode.paused or ctx.profile is None or not llm.api_key_configured():
        return []
    if job_id is not None:
        jobs, room = [job] if (job := session.get(Job, job_id)) else [], 1
    else:
        pending = session.exec(select(func.count()).select_from(Email).where(Email.status == EmailStatus.DRAFT)).one()
        room = config.caps.emails_per_day - pending  # never pile up more drafts than a day's sending
        if room <= 0:
            return []
        jobs = session.exec(select(Job).where(
            Job.state.in_(OUTREACH_STATES), Job.score >= config.matching.min_score,
            Job.outreach_note.is_(None), Job.first_seen_at >= utcnow() - OUTREACH_MAX_AGE,
        ).order_by(Job.score.desc())).all()

    system = compose.system_prompt(ctx.profile)
    drafted: list[Email] = []
    for job in jobs:
        if len(drafted) >= room:
            break
        if guards.budget_left_usd(config) <= 0:
            tally.note(f"Daily Claude budget (${config.caps.usd_per_day}) reached; rest waits.")
            break
        company_key = norm_company(job.company) or f"job-{job.id}"
        cooldown = utcnow() - timedelta(days=config.outreach.company_cooldown_days)
        recent = session.exec(select(Email).where(
            Email.company_key == company_key, Email.created_at >= cooldown,
            Email.status.in_([EmailStatus.DRAFT, EmailStatus.SENT]))).first()
        if blocklisted(job.company, config):
            job.outreach_note = "company is on your blocklist"
        elif recent is not None:
            job.outreach_note = f"already emailed {job.company} within {config.outreach.company_cooldown_days} days"
        else:
            try:
                contact, job.outreach_note = contacts.best_contact(job, allow_lookup=config.outreach.find_website)
            except llm.LLMError as e:
                contact, job.outreach_note = None, f"website lookup failed: {e}"
            if contact is not None and (why := outbox.is_suppressed(session, contact.address)):
                contact, job.outreach_note = None, f"{contact.address} is suppressed ({why})"
            if contact is not None:
                try:
                    email = compose.write(
                        job, ctx.profile, ctx.answers, system=system, to_address=contact.address,
                        source=contact.source, company_key=company_key,
                        attachment=settings.stored_path(tailor.resume_for_job(job, ctx.profile)),
                    )
                except llm.LLMError as e:
                    job.outreach_note += f"; drafting failed: {e}"
                else:
                    session.add(email)
                    drafted.append(email)
                    log(f"Drafted email to {contact.address} about {job.title} at {job.company}")
        session.commit()
    tally.drafted += len(drafted)
    return drafted


def _real_profile() -> Profile | None:
    if not settings.PROFILE_PATH.exists():
        return None
    profile = load_profile(settings.PROFILE_PATH)
    return None if profile.is_placeholder() else profile


def apply_only(*, job_id: int | None = None, live: bool = False, log: Callable[[str], None] = print) -> Tally:
    """Just the Naukri apply stage (no discovery/scoring): `jobagent apply`.

    live=True submits for real even though config.yaml says dry_run, for one job only, so the user can
    watch a first application. Pause and the setup checks still apply.
    """
    if live and job_id is None:
        raise ValueError("live override is for a single job only")
    config = load_config()
    if live:
        config = config.model_copy(deep=True)
        config.mode.dry_run = False
    ctx = MatchContext(config=config, profile=_real_profile(), answers=load_answers())
    tally = Tally()
    with db.session() as session:
        apply_naukri(session, ctx, tally, job_id=job_id, log=log)
    return tally


def run(*, smoke: bool = False, log: Callable[[str], None] = print) -> Run | None:
    config = load_config()
    if config.mode.paused:
        log("Paused (mode.paused in config.yaml): nothing done.")
        return None

    started, spent_before = utcnow(), guards.spent_today_usd()
    batch = Batch()
    searches = len(jobspy_src.queries(config.search, smoke))
    log(f"Searching {', '.join(config.search.sites)}: {searches} searches each…")
    jobspy_src.discover(config.search, batch, smoke)
    if not smoke:
        ats_boards.discover(load_targets(), config.search, batch)

    tally = Tally(errors=list(batch.errors))
    tally.notes += [f"No jobs from {site} this run." for site in config.search.sites if not batch.counts[site]]
    ctx = MatchContext(config=config, profile=_real_profile(), answers=load_answers())

    with db.session() as session:
        new, duplicates = store(session, batch.jobs)
        log(f"Found {len(batch.jobs)} ({new} new, {duplicates} duplicates). Filtering and scoring…")
        filter_discovered(session, ctx, tally)
        score_filtered(session, ctx, tally)
        tailor_scored(session, ctx, tally)
        apply_naukri(session, ctx, tally, log=log)
        draft_outreach(session, ctx, tally, log=log)
        record = Run(
            started_at=started,
            finished_at=utcnow(),
            found=len(batch.jobs),
            new=new,
            duplicates=duplicates,
            skipped=tally.skipped,
            scored=tally.scored,
            matches=tally.matches,
            tailored=tally.tailored,
            applied=tally.applied,
            drafted=tally.drafted,
            cost_usd=round(guards.spent_today_usd() - spent_before, 4),
            source_counts=dict(batch.counts),
            errors=tally.errors,
            notes=tally.notes,
        )
        session.add(record)
        session.commit()
    return record
